-- Supabase auth-backed profile and chat history schema for DocuMed AI

create extension if not exists pgcrypto;

create or replace function public.set_current_timestamp_updated_at()
returns trigger as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$ language plpgsql;

create table if not exists public.user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null unique,
  full_name text,
  avatar_url text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  last_login_at timestamptz
);

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.user_profiles(id) on delete cascade,
  title text not null,
  last_message_preview text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  render_mode text default 'plain' check (render_mode in ('plain', 'query')),
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists user_profiles_email_idx on public.user_profiles(email);
create index if not exists chat_sessions_user_id_updated_at_idx on public.chat_sessions(user_id, updated_at desc);
create index if not exists chat_messages_session_id_created_at_idx on public.chat_messages(session_id, created_at asc);

drop trigger if exists user_profiles_set_updated_at on public.user_profiles;
create trigger user_profiles_set_updated_at
before update on public.user_profiles
for each row
execute function public.set_current_timestamp_updated_at();

drop trigger if exists chat_sessions_set_updated_at on public.chat_sessions;
create trigger chat_sessions_set_updated_at
before update on public.chat_sessions
for each row
execute function public.set_current_timestamp_updated_at();

alter table public.user_profiles enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

drop policy if exists "Profiles are viewable by owner" on public.user_profiles;
create policy "Profiles are viewable by owner"
on public.user_profiles
for select
using (auth.uid() = id);

drop policy if exists "Profiles are insertable by owner" on public.user_profiles;
create policy "Profiles are insertable by owner"
on public.user_profiles
for insert
with check (auth.uid() = id);

drop policy if exists "Profiles are updatable by owner" on public.user_profiles;
create policy "Profiles are updatable by owner"
on public.user_profiles
for update
using (auth.uid() = id)
with check (auth.uid() = id);

drop policy if exists "Sessions are viewable by owner" on public.chat_sessions;
create policy "Sessions are viewable by owner"
on public.chat_sessions
for select
using (auth.uid() = user_id);

drop policy if exists "Sessions are insertable by owner" on public.chat_sessions;
create policy "Sessions are insertable by owner"
on public.chat_sessions
for insert
with check (auth.uid() = user_id);

drop policy if exists "Sessions are updatable by owner" on public.chat_sessions;
create policy "Sessions are updatable by owner"
on public.chat_sessions
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "Sessions are deletable by owner" on public.chat_sessions;
create policy "Sessions are deletable by owner"
on public.chat_sessions
for delete
using (auth.uid() = user_id);

drop policy if exists "Messages are viewable by session owner" on public.chat_messages;
create policy "Messages are viewable by session owner"
on public.chat_messages
for select
using (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = chat_messages.session_id
      and public.chat_sessions.user_id = auth.uid()
  )
);

drop policy if exists "Messages are insertable by session owner" on public.chat_messages;
create policy "Messages are insertable by session owner"
on public.chat_messages
for insert
with check (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = chat_messages.session_id
      and public.chat_sessions.user_id = auth.uid()
  )
);

drop policy if exists "Messages are deletable by session owner" on public.chat_messages;
create policy "Messages are deletable by session owner"
on public.chat_messages
for delete
using (
  exists (
    select 1
    from public.chat_sessions
    where public.chat_sessions.id = chat_messages.session_id
      and public.chat_sessions.user_id = auth.uid()
  )
);

create or replace function public.handle_new_user_profile()
returns trigger as $$
begin
  insert into public.user_profiles (id, email, full_name, avatar_url)
  values (
    new.id,
    new.email,
    new.raw_user_meta_data ->> 'full_name',
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do update
  set
    email = excluded.email,
    full_name = coalesce(excluded.full_name, public.user_profiles.full_name),
    avatar_url = coalesce(excluded.avatar_url, public.user_profiles.avatar_url),
    updated_at = timezone('utc', now());

  return new;
end;
$$ language plpgsql security definer
set search_path = public;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_user_profile();

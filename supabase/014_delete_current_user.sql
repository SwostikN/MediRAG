-- Self-service account deletion RPC.
--
-- The Supabase client library cannot delete a row from auth.users
-- directly — auth schema is owner-only. This security-definer function
-- deletes the currently-authenticated user, and cascades flow:
--   auth.users → user_profiles → chat_sessions → chat_messages
--   auth.users → session_documents → session_chunks, user_lab_markers
-- via the `on delete cascade` FKs in migrations 002–013.
--
-- Safety: the function accepts no arguments and only operates on
-- auth.uid(). A non-authenticated caller (auth.uid() IS NULL) gets a
-- no-op — it will not error and will not wipe anything. A signed-in
-- caller can only delete their own account.

create or replace function public.delete_current_user()
returns void
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  v_uid uuid := auth.uid();
begin
  if v_uid is null then
    raise exception 'delete_current_user: no authenticated user'
      using errcode = '42501';
  end if;

  -- Cascade handles the dependent rows. We delete from auth.users
  -- last so the FKs resolve correctly.
  delete from auth.users where id = v_uid;
end;
$$;

-- Allow authenticated users to call this. Anonymous callers are blocked
-- by the auth.uid() null check above anyway, but this makes the policy
-- explicit at the grant level too.
revoke all on function public.delete_current_user() from public;
grant execute on function public.delete_current_user() to authenticated;

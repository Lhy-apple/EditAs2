/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:28:26 GMT 2023
 */

package org.apache.commons.codec.language;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.codec.language.Soundex;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Soundex_ESTest extends Soundex_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Soundex soundex0 = Soundex.US_ENGLISH;
      int int0 = soundex0.getMaxLength();
      assertEquals(4, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      int int0 = soundex0.difference("i9s", "i9s");
      assertEquals(4, soundex0.getMaxLength());
      assertEquals(4, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Soundex soundex0 = null;
      try {
        soundex0 = new Soundex((char[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.language.Soundex", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Soundex soundex0 = new Soundex("t(/pq[oN'+;.{:X=F");
      // Undeclared exception!
      try { 
        soundex0.soundex("t(/pq[oN'+;.{:X=F");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The character is not mapped: T
         //
         verifyException("org.apache.commons.codec.language.Soundex", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      soundex0.US_ENGLISH.setMaxLength((-6094));
      assertEquals(4, soundex0.getMaxLength());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      Object object0 = soundex0.encode((Object) "01230120022455012623010202");
      assertEquals(4, soundex0.getMaxLength());
      assertEquals("", object0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      try { 
        soundex0.encode((Object) soundex0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Soundex encode is not of type java.lang.String
         //
         verifyException("org.apache.commons.codec.language.Soundex", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      String string0 = soundex0.soundex("cS-Vh#XM1MgJ2sY");
      assertNotNull(string0);
      assertEquals(4, soundex0.getMaxLength());
      assertEquals("C125", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      String string0 = soundex0.soundex("cw4lD&5IzPVp*^!");
      assertEquals("C432", string0);
      assertEquals(4, soundex0.getMaxLength());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      String string0 = soundex0.soundex("ZpHvX1&vZ+'2e&D0A");
      assertNotNull(string0);
      assertEquals("Z121", string0);
      assertEquals(4, soundex0.getMaxLength());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      String string0 = soundex0.soundex("*Ro\"hH91S");
      assertEquals(4, soundex0.getMaxLength());
      assertNotNull(string0);
      assertEquals("R000", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      String string0 = soundex0.soundex("&w~-hT%oeH{@J");
      assertEquals(4, soundex0.getMaxLength());
      assertEquals("W200", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Soundex soundex0 = new Soundex();
      soundex0.soundex((String) null);
      assertEquals(4, soundex0.getMaxLength());
  }
}
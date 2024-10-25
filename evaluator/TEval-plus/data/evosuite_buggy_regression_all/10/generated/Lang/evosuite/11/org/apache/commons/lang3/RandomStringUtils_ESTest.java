/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:07:16 GMT 2023
 */

package org.apache.commons.lang3;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.lang3.RandomStringUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RandomStringUtils_ESTest extends RandomStringUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String string0 = RandomStringUtils.randomAscii(3579);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      char[] charArray0 = new char[3];
      // Undeclared exception!
      try { 
        RandomStringUtils.random(3091, 492, 3091, true, true, charArray0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 492
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomAlphanumeric(3590);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String string0 = RandomStringUtils.random(128);
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.randomNumeric(3536);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      RandomStringUtils randomStringUtils0 = new RandomStringUtils();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      String string0 = RandomStringUtils.randomAlphabetic(0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        RandomStringUtils.randomAlphabetic((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requested random string length -1 is less than 0.
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      char[] charArray0 = new char[0];
      // Undeclared exception!
      try { 
        RandomStringUtils.random(3536, charArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The chars array must not be empty
         //
         verifyException("org.apache.commons.lang3.RandomStringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = RandomStringUtils.random(3590, 680, 14, true, false);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String string0 = RandomStringUtils.random(492, 2790, 492, true, true);
      assertEquals("\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6\u0AE6", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = RandomStringUtils.random(3540, 57343, 3540, false, false);
      // Undeclared exception!
      RandomStringUtils.random(55296, string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String string0 = RandomStringUtils.random(3590, 57355, 3590, false, false);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(1755, 57343, (int) '&', false, false);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      RandomStringUtils.random(579, 55296, 579, false, false);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String string0 = RandomStringUtils.random(3523, (String) null);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = RandomStringUtils.random(1, (char[]) null);
      assertEquals("\u0000", string0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:47:45 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.compress.archivers.tar.TarUtils;
import org.apache.commons.compress.archivers.zip.ZipEncoding;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarUtils_ESTest extends TarUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((byte)0, byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes(" byte binary numZber", byteArray0, (int) (byte) (-10), (int) (byte) (-10));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes((-1), byteArray0, (-1), (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1=1777777777777777777777 will not fit in octal number buffer of length -3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      TarUtils.formatLongOctalOrBinaryBytes((-576L), byteArray0, 2, (byte)26);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (byte)26, 28);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -3 at offset 0 in '\uFFFD\uFFFD{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}' len=28
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes(275L, (byte[]) null, 62, 62);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, 8);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("3]", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertEquals(0, int0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Length 0 must be at least 2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      long long0 = TarUtils.parseOctal(byteArray0, (byte)32, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)32;
      byteArray0[1] = (byte)32;
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[174];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((byte)32, byteArray0, (byte)32, (byte)32);
      assertEquals(64, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, (byte)32, 64);
      assertEquals(32L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[2] = (byte)56;
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 2, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 56 at offset 0 in '8{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-576L), byteArray0, 2, (byte)26);
      assertEquals(28, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 2, 19);
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      TarUtils.formatLongOctalOrBinaryBytes((-576L), byteArray0, 2, 9);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 9, (byte)32);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 9, 32 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[0] = (byte) (-55);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-55));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[6] = (byte) (-126);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 6, 6);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1), byteArray0, 2, (-1));
      assertEquals(1, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 2, (-1));
      assertEquals(0L, long0);
      assertArrayEquals(new byte[] {(byte) (-1), (byte)0, (byte) (-1), (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-576L), byteArray0, 2, (byte)26);
      assertEquals(28, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)26, 9);
      assertEquals((-4611686018427387904L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[6] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, 6);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[0] = (byte) (-91);
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, 8);
      assertEquals("\uFFFD", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, 12, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[25];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(1099511627776L, byteArray0, (byte)32, 8);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 39
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[165];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, 52, 52);
      assertEquals(104, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-31), byteArray0, (-31), (-31));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -31 is too large for -31 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[60];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[177];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-3994L), byteArray0, 7, 144);
      assertEquals(151, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[165];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(12, byteArray0, 148, 12);
      assertEquals(160, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte)95;
      byteArray0[1] = (byte) (-108);
      byteArray0[2] = (byte)9;
      byteArray0[3] = (byte)86;
      byteArray0[4] = (byte) (-84);
      byteArray0[5] = (byte)2;
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}
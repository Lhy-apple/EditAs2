/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:30:40 GMT 2023
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
      byte[] byteArray0 = new byte[5];
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(8589934578L, byteArray0, 0, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 8589934578=77777777762 will not fit in octal number buffer of length -2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[2] = (byte)99;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 99 at offset 0 in 'c{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes(3716, (byte[]) null, 3716, 3716);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String string0 = TarUtils.parseName((byte[]) null, 0, 0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      byte[] byteArray0 = new byte[6];
      int int0 = TarUtils.formatNameBytes(" value", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[2] = (byte) (-9);
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertEquals("\u00F7", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[3] = (byte)2;
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)2, (byte)0}, byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
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
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      long long0 = TarUtils.parseOctal(byteArray0, (byte)2, (byte)2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      TarUtils.formatLongOctalOrBinaryBytes(4, byteArray0, 4, 4);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 4, Integer.MAX_VALUE);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2147483646
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[26];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(4, byteArray0, 4, 4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 7, 7);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[21];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(7, byteArray0, 7, 7);
      assertEquals(14, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 7, 14);
      assertEquals(7L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[2] = (byte)39;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 39 at offset 0 in ''{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[21];
      TarUtils.formatLongOctalOrBinaryBytes((-238L), byteArray0, 7, 7);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 7, 14);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 7, 14 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte) (-14);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-14));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[13];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-16L), byteArray0, 4, 4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 4, 4);
      assertEquals((-16L), long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[21];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1L), byteArray0, 7, 7);
      assertEquals(14, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 7, 14);
      assertEquals((-72057594037927936L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte) (-10), (int) (byte) (-10), zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertEquals(4, int0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
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
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[26];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(1099511627782L, byteArray0, 4, 8);
      assertEquals(12, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, 31, 31);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-7), byteArray0, (-7), (-7));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -7 is too large for -7 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(1099511627776L, byteArray0, (-1), (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1099511627776 is too large for -1 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }
}
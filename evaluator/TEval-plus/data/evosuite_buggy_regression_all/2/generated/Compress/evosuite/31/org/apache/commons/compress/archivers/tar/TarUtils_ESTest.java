/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:47:29 GMT 2023
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
      byte[] byteArray0 = new byte[169];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(14, byteArray0, 14, 154);
      assertEquals(168, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes((byte) (-4), byteArray0, (byte) (-4), (byte) (-4));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -4=1777777777777777777774 will not fit in octal number buffer of length -6
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[57];
      TarUtils.formatLongOctalOrBinaryBytes((-4L), byteArray0, 2, 2);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 2, 4);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -1 at offset 0 in '\uFFFD\uFFFD{NUL}{NUL}' len=4
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((-827L), byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -827=1777777777777777776305 will not fit in octal number buffer of length -2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, (int) (byte)0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("AF:&", byteArray0, (int) (byte)0, (int) (byte) (-110), zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals((-110), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[31];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2, byteArray0, 2, 2);
      assertEquals(4, int0);
      
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, 2, 4, zipEncoding0);
      assertEquals("2 ", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      byte[] byteArray0 = new byte[22];
      byteArray0[1] = (byte) (-108);
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, 3, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (byte)1, (byte)1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Length 1 must be at least 2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 55);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2, byteArray0, 2, 2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 3, 2);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)50, (byte)32, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[43];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2, byteArray0, 2, 2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 2, 4);
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)100;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)3);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 100 at offset 0 in 'd{NUL}{NUL}' len=3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[169];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(72057594037927936L, byteArray0, 14, 14);
      assertEquals(28, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 14, 14);
      assertEquals(72057594037927936L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[43];
      TarUtils.formatLongOctalOrBinaryBytes((-6L), byteArray0, 12, 12);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 12, 24);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 12, 24 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[24];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-43L), byteArray0, 2, 2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 2, 2);
      assertEquals((-43L), long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte) (-100);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-100));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[26];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-748L), byteArray0, 9, 9);
      assertEquals(18, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 9, 9);
      assertEquals((-748L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)103);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 9
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((byte)0, byteArray0, 48, (byte)0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 46
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[141];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8589934591L, byteArray0, 8, 8);
      assertEquals(16, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-614), byteArray0, (-614), (-614));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -614 is too large for -614 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[169];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[33];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1521L), byteArray0, 11, 11);
      assertEquals(22, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}
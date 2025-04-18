/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:13:40 GMT 2023
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
      byte[] byteArray0 = new byte[28];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8, byteArray0, 8, 8);
      assertEquals(16, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 8, 8);
      assertEquals(8L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(36, (byte[]) null, 36, 36);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[2] = (byte)106;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 106 at offset 0 in 'j{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((-909L), (byte[]) null, 444, 444);
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
      byte[] byteArray0 = new byte[9];
      String string0 = TarUtils.parseName(byteArray0, 1813, (-1465));
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("CP437", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
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
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)3);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[2] = (byte)2;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, Integer.MAX_VALUE);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2147483648
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(0L, byteArray0, 3, 1);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 3, 3);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)2;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 2 at offset 0 in '\u0002{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[28];
      TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, 8, 8);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 8, 16);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 8, 16 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[21];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1L), byteArray0, 6, 6);
      assertEquals(12, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 6, 12);
      assertEquals((-281474976710656L), long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      byteArray0[0] = (byte) (-1);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-1));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      TarUtils.formatLongOctalOrBinaryBytes((-1L), byteArray0, 3, 3);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 3, 6);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 7
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      byteArray0[0] = (byte) (-29);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-29));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      byteArray0[1] = (byte) (-6);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 1, (byte)9);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, 3, (ZipEncoding) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[1] = (byte) (-83);
      // Undeclared exception!
      try { 
        TarUtils.parseName(byteArray0, (int) (byte)0, 3, (ZipEncoding) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("0*QIlvbn13_", byteArray0, (int) (byte) (-1), (int) (byte) (-1));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, 0, (int) (byte)89, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 6
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((byte)1, byteArray0, (byte)1, (byte)1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 1=1 will not fit in octal number buffer of length 0
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-1789569705), byteArray0, (-1789569705), 1877);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-2638), byteArray0, (-2638), (-2638));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -2638 is too large for -2638 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-5277), byteArray0, (-5277), (-5277));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -5277 is too large for -5277 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }
}

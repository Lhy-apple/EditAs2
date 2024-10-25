/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:44:23 GMT 2023
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
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("S}gPz$7%~~}ZIjQ}c1+", (byte[]) null, 148, 148);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(0, byteArray0, 0, 0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)63;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 63 at offset 0 in '?{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((byte)2, byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2=2 will not fit in octal number buffer of length 0
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.parseName(byteArray0, (int) (byte)0, 2949);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 2948
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("S}gPz$7%~~}ZIjQ}1+", (byte[]) null, 148, 148, zipEncoding0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      byte[] byteArray0 = new byte[29];
      byteArray0[2] = (byte)2;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertEquals("\u0002", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      byte[] byteArray0 = new byte[26];
      byteArray0[3] = (byte)2;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
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
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[53];
      long long0 = TarUtils.parseOctal(byteArray0, 22, 22);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[67];
      TarUtils.formatLongOctalOrBinaryBytes((-1954L), byteArray0, 8, 8);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 8, Integer.MAX_VALUE);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2147483642
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      TarUtils.formatLongOctalOrBinaryBytes(0L, byteArray0, (byte)1, (byte)1);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)1, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 32 at offset 1 in ' {NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[53];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2, byteArray0, 2, 2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 2, 2);
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)2;
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 2 at offset 0 in '\u0002{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      byteArray0[0] = (byte) (-17);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-17));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[77];
      TarUtils.formatLongOctalOrBinaryBytes((-1967L), byteArray0, 8, 8);
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
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[88];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1967L), byteArray0, 8, 8);
      assertEquals(16, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 8, 8);
      assertEquals((-1967L), long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[77];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(1099511627764L, byteArray0, 14, 14);
      assertEquals(28, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 14, 14);
      assertEquals(1099511627764L, long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[69];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2705L), byteArray0, 34, 34);
      assertEquals(68, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 34, 34);
      assertEquals((-2705L), long0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, 0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      byte[] byteArray0 = new byte[26];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)2, (int) (byte)2, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      int int0 = TarUtils.formatNameBytes("", byteArray0, 0, (int) (byte) (-108), zipEncoding0);
      assertEquals((-108), int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("S}gPz$7%~~}ZIjQ}1+", (byte[]) null, 100, (-1194), zipEncoding0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)2, 8, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[87];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-911), byteArray0, (-911), (-911));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -911 is too large for -911 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[59];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(1099511627903L, byteArray0, 8, 8);
      assertEquals(16, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[37];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[67];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[170];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-138L), byteArray0, 76, 76);
      assertEquals(152, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = new byte[179];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2035L, byteArray0, 79, 79);
      assertEquals(158, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }
}

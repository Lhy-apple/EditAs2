/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:30:39 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.Instant;
import java.time.temporal.ChronoField;
import java.util.Date;
import org.apache.commons.compress.archivers.zip.X5455_ExtendedTimestamp;
import org.apache.commons.compress.archivers.zip.ZipLong;
import org.apache.commons.compress.archivers.zip.ZipShort;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.time.MockInstant;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class X5455_ExtendedTimestamp_ESTest extends X5455_ExtendedTimestamp_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      boolean boolean0 = x5455_ExtendedTimestamp0.isBit0_modifyTimePresent();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = x5455_ExtendedTimestamp0.getModifyTime();
      assertNull(zipLong0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = x5455_ExtendedTimestamp0.getCreateTime();
      assertNull(zipLong0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte byte0 = x5455_ExtendedTimestamp0.getFlags();
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      Date date0 = x5455_ExtendedTimestamp0.getAccessJavaTime();
      x5455_ExtendedTimestamp0.setAccessJavaTime(date0);
      assertFalse(x5455_ExtendedTimestamp0.isBit1_accessTimePresent());
      assertEquals((byte)0, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      boolean boolean0 = x5455_ExtendedTimestamp0.isBit1_accessTimePresent();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      boolean boolean0 = x5455_ExtendedTimestamp0.isBit2_createTimePresent();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setModifyJavaTime((Date) null);
      assertEquals((byte)0, x5455_ExtendedTimestamp0.getFlags());
      assertFalse(x5455_ExtendedTimestamp0.isBit0_modifyTimePresent());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.AED_SIG;
      x5455_ExtendedTimestamp0.setModifyTime(zipLong0);
      x5455_ExtendedTimestamp0.toString();
      assertEquals((byte)1, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      Date date0 = x5455_ExtendedTimestamp0.getCreateJavaTime();
      assertNull(date0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipShort zipShort0 = x5455_ExtendedTimestamp0.getHeaderId();
      assertEquals(21589, zipShort0.getValue());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = x5455_ExtendedTimestamp0.getAccessTime();
      assertNull(zipLong0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      Object object0 = x5455_ExtendedTimestamp0.clone();
      assertNotSame(object0, x5455_ExtendedTimestamp0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setFlags((byte)15);
      // Undeclared exception!
      try { 
        x5455_ExtendedTimestamp0.getCentralDirectoryData();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.X5455_ExtendedTimestamp", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipLong zipLong0 = ZipLong.DD_SIG;
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      x5455_ExtendedTimestamp0.getLocalFileDataData();
      assertTrue(x5455_ExtendedTimestamp0.isBit1_accessTimePresent());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.ZIP64_MAGIC;
      x5455_ExtendedTimestamp0.setCreateTime(zipLong0);
      x5455_ExtendedTimestamp0.getCentralDirectoryData();
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setFlags((byte)30);
      x5455_ExtendedTimestamp0.getCentralDirectoryData();
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte[] byteArray0 = new byte[3];
      byteArray0[2] = (byte)1;
      // Undeclared exception!
      try { 
        x5455_ExtendedTimestamp0.parseFromCentralDirectoryData(byteArray0, (byte)2, (byte)1);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3
         //
         verifyException("org.apache.commons.compress.utils.ByteUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte[] byteArray0 = new byte[5];
      byteArray0[4] = (byte)2;
      // Undeclared exception!
      try { 
        x5455_ExtendedTimestamp0.parseFromCentralDirectoryData(byteArray0, (byte)4, 3340);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 5
         //
         verifyException("org.apache.commons.compress.utils.ByteUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte[] byteArray0 = new byte[5];
      byteArray0[2] = (byte)2;
      x5455_ExtendedTimestamp0.parseFromLocalFileData(byteArray0, (byte)2, (byte)1);
      assertEquals((byte)2, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte[] byteArray0 = new byte[8];
      byteArray0[4] = (byte)4;
      // Undeclared exception!
      try { 
        x5455_ExtendedTimestamp0.parseFromCentralDirectoryData(byteArray0, (byte)4, 31);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("org.apache.commons.compress.utils.ByteUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      byte[] byteArray0 = new byte[3];
      byteArray0[2] = (byte)4;
      x5455_ExtendedTimestamp0.parseFromCentralDirectoryData(byteArray0, (byte)2, (byte)4);
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setCreateJavaTime((Date) null);
      assertFalse(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
      assertEquals((byte)0, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      MockDate mockDate0 = new MockDate(4911, (byte)2, (byte)1, 4911, (byte)1, (byte)1);
      // Undeclared exception!
      try { 
        x5455_ExtendedTimestamp0.setAccessJavaTime(mockDate0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // X5455 timestamps must fit in a signed 32 bit integer: 152789986861
         //
         verifyException("org.apache.commons.compress.archivers.zip.X5455_ExtendedTimestamp", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setFlags((byte)1);
      x5455_ExtendedTimestamp0.toString();
      assertTrue(x5455_ExtendedTimestamp0.isBit0_modifyTimePresent());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = new ZipLong((long) (byte)1);
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      x5455_ExtendedTimestamp0.toString();
      assertTrue(x5455_ExtendedTimestamp0.isBit1_accessTimePresent());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setFlags((byte)2);
      x5455_ExtendedTimestamp0.toString();
      assertEquals((byte)2, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      x5455_ExtendedTimestamp0.setFlags((byte)4);
      x5455_ExtendedTimestamp0.toString();
      assertEquals((byte)4, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = new ZipLong(28789);
      x5455_ExtendedTimestamp0.setCreateTime(zipLong0);
      x5455_ExtendedTimestamp0.toString();
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ChronoField chronoField0 = ChronoField.INSTANT_SECONDS;
      boolean boolean0 = x5455_ExtendedTimestamp0.equals(chronoField0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.AED_SIG;
      x5455_ExtendedTimestamp0.setCreateTime(zipLong0);
      boolean boolean0 = x5455_ExtendedTimestamp1.equals(x5455_ExtendedTimestamp0);
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.SINGLE_SEGMENT_SPLIT_MARKER;
      x5455_ExtendedTimestamp1.setModifyTime(zipLong0);
      x5455_ExtendedTimestamp0.setFlags((byte)1);
      boolean boolean0 = x5455_ExtendedTimestamp1.equals(x5455_ExtendedTimestamp0);
      assertTrue(x5455_ExtendedTimestamp1.isBit0_modifyTimePresent());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.AED_SIG;
      x5455_ExtendedTimestamp0.setModifyTime(zipLong0);
      x5455_ExtendedTimestamp1.setFlags((byte)1);
      boolean boolean0 = x5455_ExtendedTimestamp1.equals(x5455_ExtendedTimestamp0);
      assertEquals((byte)1, x5455_ExtendedTimestamp1.getFlags());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.LFH_SIG;
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      Date date0 = x5455_ExtendedTimestamp0.getAccessJavaTime();
      x5455_ExtendedTimestamp1.setAccessJavaTime(date0);
      boolean boolean0 = x5455_ExtendedTimestamp0.equals(x5455_ExtendedTimestamp1);
      assertTrue(x5455_ExtendedTimestamp1.isBit1_accessTimePresent());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.ZIP64_MAGIC;
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      x5455_ExtendedTimestamp1.setFlags((byte)2);
      boolean boolean0 = x5455_ExtendedTimestamp1.equals(x5455_ExtendedTimestamp0);
      assertEquals((byte)2, x5455_ExtendedTimestamp1.getFlags());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.ZIP64_MAGIC;
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      x5455_ExtendedTimestamp1.setFlags((byte)2);
      boolean boolean0 = x5455_ExtendedTimestamp0.equals(x5455_ExtendedTimestamp1);
      assertTrue(x5455_ExtendedTimestamp0.isBit1_accessTimePresent());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.ZIP64_MAGIC;
      x5455_ExtendedTimestamp0.setCreateTime(zipLong0);
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      Instant instant0 = MockInstant.ofEpochSecond(4294967296L, (-493L));
      Date date0 = Date.from(instant0);
      x5455_ExtendedTimestamp1.setCreateJavaTime(date0);
      boolean boolean0 = x5455_ExtendedTimestamp0.equals(x5455_ExtendedTimestamp1);
      assertTrue(x5455_ExtendedTimestamp1.isBit2_createTimePresent());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.AED_SIG;
      x5455_ExtendedTimestamp1.setCreateTime(zipLong0);
      x5455_ExtendedTimestamp0.setFlags((byte)4);
      boolean boolean0 = x5455_ExtendedTimestamp0.equals(x5455_ExtendedTimestamp1);
      assertTrue(x5455_ExtendedTimestamp1.isBit2_createTimePresent());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp1 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.AED_SIG;
      x5455_ExtendedTimestamp1.setCreateTime(zipLong0);
      x5455_ExtendedTimestamp0.setFlags((byte)4);
      boolean boolean0 = x5455_ExtendedTimestamp1.equals(x5455_ExtendedTimestamp0);
      assertTrue(x5455_ExtendedTimestamp1.isBit2_createTimePresent());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.DD_SIG;
      x5455_ExtendedTimestamp0.setModifyTime(zipLong0);
      x5455_ExtendedTimestamp0.hashCode();
      assertEquals((byte)1, x5455_ExtendedTimestamp0.getFlags());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.CFH_SIG;
      x5455_ExtendedTimestamp0.setAccessTime(zipLong0);
      x5455_ExtendedTimestamp0.hashCode();
      assertTrue(x5455_ExtendedTimestamp0.isBit1_accessTimePresent());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      X5455_ExtendedTimestamp x5455_ExtendedTimestamp0 = new X5455_ExtendedTimestamp();
      ZipLong zipLong0 = ZipLong.ZIP64_MAGIC;
      x5455_ExtendedTimestamp0.setCreateTime(zipLong0);
      x5455_ExtendedTimestamp0.hashCode();
      assertTrue(x5455_ExtendedTimestamp0.isBit2_createTimePresent());
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:21:57 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import org.apache.commons.compress.archivers.zip.AsiExtraField;
import org.apache.commons.compress.archivers.zip.GeneralPurposeBit;
import org.apache.commons.compress.archivers.zip.UnicodeCommentExtraField;
import org.apache.commons.compress.archivers.zip.UnicodePathExtraField;
import org.apache.commons.compress.archivers.zip.UnparseableExtraFieldData;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipExtraField;
import org.apache.commons.compress.archivers.zip.ZipShort;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveEntry_ESTest extends ZipArchiveEntry_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getLastModifiedDate();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipExtraField[] zipExtraFieldArray0 = zipArchiveEntry0.getExtraFields();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipExtraFieldArray0.length);
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("Svf(%(gGII1E{L3G");
      zipArchiveEntry0.hashCode();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = null;
      try {
        zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ZIP compression method can not be negative: -1
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getUnparseableExtraFieldData();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getGeneralPurposeBit();
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      zipArchiveEntry0.setUnixMode(3);
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(196609L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipArchiveEntry zipArchiveEntry2 = null;
      try {
        zipArchiveEntry2 = new ZipArchiveEntry(zipArchiveEntry1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // ZIP compression method can not be negative: -1
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipUtil");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "JarMarker doesn't expect any data");
      byte[] byteArray0 = new byte[6];
      byteArray0[3] = (byte)7;
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.removeUnparseableExtraFieldData();
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("/", "/");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "/");
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipUtil");
      mockFile0.mkdirs();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "org.apache.commons.compress.archivers.zip.ZipUtil");
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("og.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      zipArchiveEntry0.setUnixMode(128);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("/");
      zipArchiveEntry0.setUnixMode(0);
      assertEquals(3, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("/");
      int int0 = zipArchiveEntry0.getUnixMode();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, int0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)123;
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnparseableExtraFieldData unparseableExtraFieldData0 = new UnparseableExtraFieldData();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.addAsFirstExtraField(unparseableExtraFieldData0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ExtraFieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry1.getExtraFields();
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      byte[] byteArray0 = new byte[5];
      zipArchiveEntry1.setExtra(byteArray0);
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnparseableExtraFieldData unparseableExtraFieldData0 = new UnparseableExtraFieldData();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.addExtraField(unparseableExtraFieldData0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ExtraFieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      AsiExtraField asiExtraField0 = new AsiExtraField();
      zipArchiveEntry0.addAsFirstExtraField(asiExtraField0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      AsiExtraField asiExtraField0 = new AsiExtraField();
      zipArchiveEntry1.addAsFirstExtraField(asiExtraField0);
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      byte[] byteArray0 = new byte[1];
      UnicodePathExtraField unicodePathExtraField0 = new UnicodePathExtraField("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", byteArray0);
      zipArchiveEntry0.addExtraField(unicodePathExtraField0);
      zipArchiveEntry0.removeExtraField(unicodePathExtraField0.UPATH_ID);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipShort zipShort0 = UnicodeCommentExtraField.UCOM_ID;
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(zipShort0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      AsiExtraField asiExtraField0 = new AsiExtraField();
      zipArchiveEntry0.addExtraField(asiExtraField0);
      ZipShort zipShort0 = asiExtraField0.getLocalFileDataLength();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField(zipShort0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeUnparseableExtraFieldData();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      AsiExtraField asiExtraField0 = new AsiExtraField();
      ZipShort zipShort0 = asiExtraField0.getHeaderId();
      zipArchiveEntry0.getExtraField(zipShort0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setName((String) null);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertTrue(boolean0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.setSize((-154L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // invalid entry size
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getRawName();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[2];
      zipArchiveEntry0.setName("/", byteArray0);
      zipArchiveEntry0.getRawName();
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipUtil", "org.apache.commons.compress.archivers.zip.ZipUtil");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "JaMarker doesn't expect any data");
      byte[] byteArray0 = new byte[6];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertTrue(boolean0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals((Object) null);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipUtil");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "JarMarker doesn't expect any data");
      boolean boolean0 = zipArchiveEntry0.equals((byte)7);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry("LAjBo(:==");
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertFalse(boolean0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("00KH.");
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertTrue(boolean0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("~+eyddQFXm|,JE(P");
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("] for UniCode path extra data./");
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1), zipArchiveEntry1.getMethod());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setTime(3);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      zipArchiveEntry1.setInternalAttributes(3);
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(3, zipArchiveEntry1.getInternalAttributes());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setPlatform(3);
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(boolean0);
      assertEquals(3, zipArchiveEntry1.getPlatform());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setExternalAttributes(3);
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(3L, zipArchiveEntry1.getExternalAttributes());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setMethod(3);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(3, zipArchiveEntry1.getMethod());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setSize(0);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(0L, zipArchiveEntry1.getSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setCrc(3);
      zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setCompressedSize(3);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setGeneralPurposeBit((GeneralPurposeBit) null);
      zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }
}

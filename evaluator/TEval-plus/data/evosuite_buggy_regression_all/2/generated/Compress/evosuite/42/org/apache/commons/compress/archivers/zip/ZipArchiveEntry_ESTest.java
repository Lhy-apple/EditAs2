/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:48:13 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.NoSuchElementException;
import org.apache.commons.compress.archivers.zip.AsiExtraField;
import org.apache.commons.compress.archivers.zip.UnicodePathExtraField;
import org.apache.commons.compress.archivers.zip.UnparseableExtraFieldData;
import org.apache.commons.compress.archivers.zip.X0014_X509Certificates;
import org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField;
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
      int int0 = zipArchiveEntry0.getVersionRequired();
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, int0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipExtraField[] zipExtraFieldArray0 = zipArchiveEntry1.getExtraFields(true);
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipExtraFieldArray0.length);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(3);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry(zipArchiveEntry0);
      assertEquals(3, zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getLastModifiedDate();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getExtraFields();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertTrue(boolean0);
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("h");
      zipArchiveEntry0.hashCode();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getUnparseableExtraFieldData();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setExtra(byteArray0);
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setCentralDirectoryExtra(byteArray0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[0];
      zipArchiveEntry0.setName((String) null, byteArray0);
      boolean boolean0 = zipArchiveEntry0.isDirectory();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setRawFlag((-1));
      assertEquals((-1), zipArchiveEntry0.getRawFlag());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setVersionMadeBy(0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getVersionMadeBy());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      int int0 = zipArchiveEntry0.getVersionMadeBy();
      assertEquals(0, int0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setVersionRequired((-1));
      assertEquals((-1), zipArchiveEntry0.getVersionRequired());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("h");
      int int0 = zipArchiveEntry0.getRawFlag();
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, int0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(" ");
      byte[] byteArray0 = new byte[7];
      zipArchiveEntry0.setExtra(byteArray0);
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
  public void test17()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "L28bL(");
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, ">ycV1+'e@zN;=Oj f/");
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry("dcJ@RH?}/H/");
      zipArchiveEntry0.setUnixMode(0);
      assertEquals(17L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setUnixMode((-814));
      boolean boolean0 = zipArchiveEntry0.isUnixSymlink();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.isUnixSymlink();
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getExtraFields(false);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipExtraField[] zipExtraFieldArray0 = zipArchiveEntry1.getExtraFields(false);
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertNotSame(zipArchiveEntry1, zipArchiveEntry0);
      assertEquals(0, zipExtraFieldArray0.length);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnparseableExtraFieldData unparseableExtraFieldData0 = new UnparseableExtraFieldData();
      zipArchiveEntry0.addAsFirstExtraField(unparseableExtraFieldData0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getExtraFields(true);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry1.setExtra(byteArray0);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertFalse(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnparseableExtraFieldData unparseableExtraFieldData0 = new UnparseableExtraFieldData();
      zipArchiveEntry0.addExtraField(unparseableExtraFieldData0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      X0014_X509Certificates x0014_X509Certificates0 = new X0014_X509Certificates();
      zipArchiveEntry0.addExtraField(x0014_X509Certificates0);
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      UnicodePathExtraField unicodePathExtraField0 = new UnicodePathExtraField();
      zipArchiveEntry0.addAsFirstExtraField(unicodePathExtraField0);
      zipArchiveEntry0.addExtraField(unicodePathExtraField0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      X0014_X509Certificates x0014_X509Certificates0 = new X0014_X509Certificates();
      zipArchiveEntry0.addAsFirstExtraField(x0014_X509Certificates0);
      zipArchiveEntry0.addAsFirstExtraField(x0014_X509Certificates0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.removeExtraField((ZipShort) null);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      X0014_X509Certificates x0014_X509Certificates0 = new X0014_X509Certificates();
      byte[] byteArray0 = new byte[4];
      zipArchiveEntry0.setExtra(byteArray0);
      zipArchiveEntry0.addAsFirstExtraField(x0014_X509Certificates0);
      zipArchiveEntry0.addAsFirstExtraField(x0014_X509Certificates0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = (ZipArchiveEntry)zipArchiveEntry0.clone();
      ZipShort zipShort0 = Zip64ExtendedInformationExtraField.HEADER_ID;
      // Undeclared exception!
      try { 
        zipArchiveEntry1.removeExtraField(zipShort0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(" ");
      byte[] byteArray0 = new byte[7];
      byteArray0[2] = (byte)48;
      zipArchiveEntry0.setExtra(byteArray0);
      zipArchiveEntry0.removeUnparseableExtraFieldData();
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
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
  public void test36()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setPlatform(3313);
      zipArchiveEntry0.setName("gd5v,9i-;'H|3");
      assertEquals(3313, zipArchiveEntry0.getPlatform());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      // Undeclared exception!
      try { 
        zipArchiveEntry0.setSize((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // invalid entry size
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.getRawName();
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      MockFile mockFile0 = new MockFile("dgs<1L{ZHx", "invalid entry size");
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry(mockFile0, "invalid entry size");
      byte[] byteArray0 = new byte[0];
      zipArchiveEntry0.setName((String) null, byteArray0);
      zipArchiveEntry0.getRawName();
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getSize());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)8;
      zipArchiveEntry0.setExtra(byteArray0);
      zipArchiveEntry0.setExtra(byteArray0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertTrue(boolean0);
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals((Object) null);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      AsiExtraField asiExtraField0 = new AsiExtraField();
      boolean boolean0 = zipArchiveEntry0.equals(asiExtraField0);
      assertEquals((-1), zipArchiveEntry0.getMethod());
      assertEquals(0, zipArchiveEntry0.getPlatform());
      assertFalse(boolean0);
      assertEquals(0L, zipArchiveEntry0.getExternalAttributes());
      assertEquals((-1L), zipArchiveEntry0.getSize());
      assertEquals(0, zipArchiveEntry0.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry("g`E'+Q");
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertFalse(boolean0);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("*Md>{");
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(boolean0);
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertFalse(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setComment("2:YNg}Tl\"");
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertFalse(boolean0);
      assertFalse(zipArchiveEntry0.equals((Object)zipArchiveEntry1));
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1L), zipArchiveEntry1.getSize());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setTime(380L);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setInternalAttributes((-1));
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals((-1), zipArchiveEntry1.getInternalAttributes());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setUnixMode((-1));
      zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(3, zipArchiveEntry1.getPlatform());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      
      zipArchiveEntry1.setExternalAttributes((-1));
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setMethod(0);
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertEquals(0, zipArchiveEntry0.getMethod());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.setSize(3);
      boolean boolean0 = zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals(3L, zipArchiveEntry1.getSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      assertTrue(zipArchiveEntry1.equals((Object)zipArchiveEntry0));
      
      zipArchiveEntry1.setCrc(3);
      boolean boolean0 = zipArchiveEntry0.equals(zipArchiveEntry1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      zipArchiveEntry0.setCompressedSize((-3802L));
      ZipArchiveEntry zipArchiveEntry1 = new ZipArchiveEntry();
      zipArchiveEntry1.equals(zipArchiveEntry0);
      assertEquals((-1L), zipArchiveEntry1.getSize());
      assertEquals(0L, zipArchiveEntry1.getExternalAttributes());
      assertEquals(0, zipArchiveEntry1.getPlatform());
      assertEquals((-1), zipArchiveEntry1.getMethod());
      assertEquals(0, zipArchiveEntry1.getInternalAttributes());
  }
}

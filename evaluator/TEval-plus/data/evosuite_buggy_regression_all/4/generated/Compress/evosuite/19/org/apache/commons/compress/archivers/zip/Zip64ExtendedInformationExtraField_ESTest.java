/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:30:07 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField;
import org.apache.commons.compress.archivers.zip.ZipEightByteInteger;
import org.apache.commons.compress.archivers.zip.ZipLong;
import org.apache.commons.compress.archivers.zip.ZipShort;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Zip64ExtendedInformationExtraField_ESTest extends Zip64ExtendedInformationExtraField_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      zip64ExtendedInformationExtraField0.setCompressedSize(zipEightByteInteger0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      zip64ExtendedInformationExtraField0.setSize(zipEightByteInteger0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      ZipLong zipLong0 = zip64ExtendedInformationExtraField0.getDiskStartNumber();
      assertNull(zipLong0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      ZipLong zipLong0 = ZipLong.LFH_SIG;
      zip64ExtendedInformationExtraField0.setDiskStartNumber(zipLong0);
      assertEquals(67324752L, zipLong0.getValue());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      ZipEightByteInteger zipEightByteInteger0 = zip64ExtendedInformationExtraField0.getRelativeHeaderOffset();
      assertNull(zipEightByteInteger0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      ZipEightByteInteger zipEightByteInteger1 = zip64ExtendedInformationExtraField0.getCompressedSize();
      assertSame(zipEightByteInteger0, zipEightByteInteger1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      ZipShort zipShort0 = zip64ExtendedInformationExtraField0.getHeaderId();
      assertEquals(1, zipShort0.getValue());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField((ZipEightByteInteger) null, (ZipEightByteInteger) null);
      ZipEightByteInteger zipEightByteInteger0 = zip64ExtendedInformationExtraField0.getSize();
      assertNull(zipEightByteInteger0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      ZipShort zipShort0 = zip64ExtendedInformationExtraField0.getLocalFileDataLength();
      assertEquals(0, zipShort0.getValue());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      ZipShort zipShort0 = zip64ExtendedInformationExtraField0.getLocalFileDataLength();
      assertEquals(16, zipShort0.getValue());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      zip64ExtendedInformationExtraField0.parseFromLocalFileData(byteArray0, 0, 16);
      assertEquals(16, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      zip64ExtendedInformationExtraField0.setRelativeHeaderOffset(zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      assertEquals(24, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      ZipLong zipLong0 = ZipLong.DD_SIG;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0, zipEightByteInteger0, zipLong0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      assertEquals(28, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getLocalFileDataData();
      zip64ExtendedInformationExtraField0.parseFromCentralDirectoryData(byteArray0, (byte)4, (byte)4);
      zip64ExtendedInformationExtraField0.reparseCentralDirectoryData(false, false, false, true);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getLocalFileDataData();
      zip64ExtendedInformationExtraField0.parseFromLocalFileData(byteArray0, (byte)0, (byte)0);
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField((ZipEightByteInteger) null, zipEightByteInteger0);
      // Undeclared exception!
      try { 
        zip64ExtendedInformationExtraField0.getLocalFileDataData();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Zip64 extended information must contain both size values in the local file header.
         //
         verifyException("org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, (ZipEightByteInteger) null);
      // Undeclared exception!
      try { 
        zip64ExtendedInformationExtraField0.getLocalFileDataData();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Zip64 extended information must contain both size values in the local file header.
         //
         verifyException("org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getLocalFileDataData();
      try { 
        zip64ExtendedInformationExtraField0.parseFromLocalFileData(byteArray0, 8, 8);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // Zip64 extended information must contain both size values in the local file header.
         //
         verifyException("org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      // Undeclared exception!
      try { 
        zip64ExtendedInformationExtraField0.parseFromLocalFileData(byteArray0, 0, 24);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 23
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipEightByteInteger", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      // Undeclared exception!
      try { 
        zip64ExtendedInformationExtraField0.parseFromLocalFileData(byteArray0, 0, 22);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 19
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipLong", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      zip64ExtendedInformationExtraField0.parseFromCentralDirectoryData(byteArray0, 0, 16);
      zip64ExtendedInformationExtraField0.reparseCentralDirectoryData(true, true, false, false);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      zip64ExtendedInformationExtraField0.parseFromCentralDirectoryData(byteArray0, (byte)4, (byte)4);
      try { 
        zip64ExtendedInformationExtraField0.reparseCentralDirectoryData(true, true, true, true);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // central directory zip64 extended information extra field's length doesn't match central directory data.  Expected length 28 but is 4
         //
         verifyException("org.apache.commons.compress.archivers.zip.Zip64ExtendedInformationExtraField", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField();
      zip64ExtendedInformationExtraField0.reparseCentralDirectoryData(true, true, true, true);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipEightByteInteger zipEightByteInteger0 = ZipEightByteInteger.ZERO;
      Zip64ExtendedInformationExtraField zip64ExtendedInformationExtraField0 = new Zip64ExtendedInformationExtraField(zipEightByteInteger0, zipEightByteInteger0);
      byte[] byteArray0 = zip64ExtendedInformationExtraField0.getCentralDirectoryData();
      zip64ExtendedInformationExtraField0.parseFromCentralDirectoryData(byteArray0, (byte)8, (byte)8);
      zip64ExtendedInformationExtraField0.reparseCentralDirectoryData(false, false, true, false);
  }
}

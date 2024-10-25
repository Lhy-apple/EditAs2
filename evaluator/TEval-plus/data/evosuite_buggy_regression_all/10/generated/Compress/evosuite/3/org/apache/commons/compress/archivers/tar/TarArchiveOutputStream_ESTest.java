/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:06:49 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(10240, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 100);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.flush();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("6Q?y[htLCc_\u0007&~htX");
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "6Q?y[htLCc_\u0007&~htX");
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(2);
      MockFile mockFile0 = new MockFile("/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry", "/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry");
      MockFile mockFile1 = new MockFile(mockFile0, "/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry/org.apache.commons(compress.archivers.cpio.CpioArQIiveOntry");
      MockFile mockFile2 = new MockFile(mockFile1, "JarMarker doesn't expec any dta");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile2);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(0, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons(compress.archivers.cpio.CpioArchiveOntry", "org.apache.commons(compress.archivers.cpio.CpioArchiveOntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons(compress.archivers.cpio.CpioArchiveOntry", "org.apache.commons(compress.archivers.cpio.CpioArchiveOntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(1);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(16877, TarArchiveEntry.DEFAULT_DIR_MODE);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "");
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(529);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 529, 529);
      MockFile mockFile0 = new MockFile("' bytes for entry '");
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("' bytes for entry '");
      byte[] byteArray0 = new byte[8];
      FileSystemHandling.appendDataToFile(evoSuiteFile0, byteArray0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry '' bytes for entry '' closed at '0' before the '8' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.close();
      try { 
        byteArrayOutputStream0.writeTo(tarArchiveOutputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '10240' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(2);
      MockFile mockFile0 = new MockFile("/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry", "/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry");
      MockFile mockFile1 = new MockFile(mockFile0, "/data/swf/zenodo_replication_package_new/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry/org.apache.commons(compress.archivers.cpio.CpioArchiveOntry");
      MockFile mockFile2 = new MockFile(mockFile1, "org.apache.commons(compress.archivers.cpio.CpioArchiveOntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile2);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(0, byteArrayOutputStream0.size());
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:32:43 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.apache.commons.compress.archivers.ArchiveEntry;
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
      tarArchiveOutputStream0.flush();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      TarArchiveOutputStream tarArchiveOutputStream1 = new TarArchiveOutputStream(tarArchiveOutputStream0, 0);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 423, 423);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(423, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1934, 1934);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(3868, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1943, 1943);
      MockFile mockFile0 = new MockFile("org.apache.commons.comress.archivers.zip.UnrecognizedExtraField", "org.apache.commons.comress.archivers.zip.UnrecognizedExtraField");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.setLongFileMode(2);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(3886, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons.comress.archivers.zip.UnrecognizedExtraField", "org.apache.commons.comress.archivers.zip.UnrecognizedExtraField");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/swf/zenodo_replication_package_new/org.apache.commons.comress.archivers.zip.UnrecognizedExtraField/org.apache.commons.comress.archivers.zip.UnrecognizedExtraField' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons.comress.archivers.zip.UnrecognizedExtraField", "org.apache.commons.comress.archivers.zip.UnrecognizedExtraField");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.setLongFileMode(1);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 423, 423);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 423, 423);
      MockFile mockFile0 = new MockFile("3#v/&[@lR_CE(QYo>", "3#v/&[@lR_CE(QYo>");
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("3#v/&[@lR_CE(QYo>/3#v/&[@lR_CE(QYo>");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "3#v/&[@lR_CE(QYo>");
      ArchiveEntry archiveEntry0 = tarArchiveOutputStream0.createArchiveEntry(mockFile0, "3#v/&[@lR_CE(QYo>");
      tarArchiveOutputStream0.putArchiveEntry(archiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry '3#v/&[@lR_CE(QYo>' closed at '0' before the '17' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      byte[] byteArray0 = new byte[1];
      try { 
        tarArchiveOutputStream0.write(byteArray0, 1, 1);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '1' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1924, 460);
      MockFile mockFile0 = new MockFile("org.apache.commons.comress.archivers.zip.UnrecognizedExtraField", "org.apache.commons.comress.archivers.zip.UnrecognizedExtraField");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "org.apache.commons.comress.archivers.zip.UnrecognizedExtraField");
      tarArchiveEntry0.setSize(65280L);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      byte[] byteArray0 = new byte[0];
      try { 
        tarArchiveOutputStream0.write(byteArray0, 443, 29127);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // record has length '0' with offset '443' which is less than the record size of '460'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }
}
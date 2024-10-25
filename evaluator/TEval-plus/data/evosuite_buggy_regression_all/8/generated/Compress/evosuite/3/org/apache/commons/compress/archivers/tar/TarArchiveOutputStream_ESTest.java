/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:39:51 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
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
  public void test01()  throws Throwable  {
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0, 0);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      MockFile mockFile0 = new MockFile(" doesn't implement ZipExtraField", " doesn't implement ZipExtraField");
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, " doesn't implement ZipExtraField");
      assertEquals(31, TarArchiveEntry.MAX_NAMELEN);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("4r-E()");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFileOutputStream0);
      DataOutputStream dataOutputStream0 = new DataOutputStream(mockPrintStream0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFile mockFile0 = new MockFile("will not fit in octal number buffer of length ", "will not fit in octal number buffer of length ");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setLongFileMode(2);
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Output buffer is closed
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockFile mockFile0 = new MockFile("FATAL: UTF-8 encoding not supported.", "FATAL: UTF-8 encoding not supported.");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/swf/zenodo_replication_package_new/FATAL: UTF-8 encoding not supported./FATAL: UTF-8 encoding not supported.' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("FATAL: UTF-8 encoding not supported.", "FATAL: UTF-8 encoding not supported.");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.setLongFileMode(1);
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Output buffer is closed
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("/", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("r\"u;4");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("r\"u;4");
      tarArchiveEntry0.setSize(2);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      tarArchiveOutputStream0.write(2);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'r\"u;4' closed at '1' before the '2' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      try { 
        tarArchiveOutputStream0.write(1);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '1' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFile mockFile0 = new MockFile("' which is less than the record size of '", "' which is less than the record size of '");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "dyJ(DiYwy*.>1ZBA");
      tarArchiveEntry0.setSize(1000);
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      tarArchiveOutputStream0.write(2);
      tarArchiveOutputStream0.write(2);
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockFile mockFile0 = new MockFile("' which is less than the record size of '", "' which is less than the record size of '");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "' which is less than the record size of '");
      tarArchiveEntry0.setSize(1000);
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      byte[] byteArray0 = new byte[3];
      tarArchiveOutputStream0.write((-944));
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.write(byteArray0, (-944), 511);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockFile mockFile0 = new MockFile("' which is less than the record size of '", "' which is less than the record size of '");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "' which is less than the record size of '");
      tarArchiveEntry0.setSize(1000);
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      byte[] byteArray0 = new byte[0];
      try { 
        tarArchiveOutputStream0.write(byteArray0, 1000, 1000);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // record has length '0' with offset '1000' which is less than the record size of '512'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarBuffer", e);
      }
  }
}

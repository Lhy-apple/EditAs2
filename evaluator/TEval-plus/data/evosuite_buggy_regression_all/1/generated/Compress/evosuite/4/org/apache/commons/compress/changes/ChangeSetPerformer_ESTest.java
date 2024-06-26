/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:24:29 GMT 2023
 */

package org.apache.commons.compress.changes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.InputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ArchiveOutputStream;
import org.apache.commons.compress.archivers.cpio.CpioArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.changes.ChangeSet;
import org.apache.commons.compress.changes.ChangeSetPerformer;
import org.apache.commons.compress.changes.ChangeSetResults;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ChangeSetPerformer_ESTest extends ChangeSetPerformer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      MockFile mockFile0 = new MockFile("kR0z!O*0[_rOZqR@");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      mockPrintStream0.append('R');
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 1426);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(mockFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0);
      ChangeSetResults changeSetResults0 = changeSetPerformer0.perform(tarArchiveInputStream0, tarArchiveOutputStream0);
      assertNotNull(changeSetResults0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("%*");
      JarArchiveInputStream jarArchiveInputStream0 = new JarArchiveInputStream((InputStream) null);
      changeSet0.add((ArchiveEntry) cpioArchiveEntry0, (InputStream) jarArchiveInputStream0);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      // Undeclared exception!
      try { 
        changeSetPerformer0.perform(jarArchiveInputStream0, (ArchiveOutputStream) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.changes.ChangeSetPerformer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      changeSet0.delete("kR0z!O*0[_rOZqR@");
      MockFile mockFile0 = new MockFile("kR0z!O*0[_rOZqR@");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      mockPrintStream0.append('R');
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 1426);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(mockFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0);
      ChangeSetResults changeSetResults0 = changeSetPerformer0.perform(tarArchiveInputStream0, tarArchiveOutputStream0);
      assertNotNull(changeSetResults0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      MockFile mockFile0 = new MockFile("kR0z!O*0[_rOZqR@");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      mockPrintStream0.append('R');
      changeSet0.deleteDir("kR0z!O*0[_rOZqR@");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 1426);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(mockFile0);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(mockFileInputStream0);
      ChangeSetResults changeSetResults0 = changeSetPerformer0.perform(tarArchiveInputStream0, tarArchiveOutputStream0);
      assertNotNull(changeSetResults0);
  }
}

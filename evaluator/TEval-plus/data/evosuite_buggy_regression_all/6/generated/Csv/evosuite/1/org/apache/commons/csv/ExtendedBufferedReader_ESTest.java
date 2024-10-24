/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:14:53 GMT 2023
 */

package org.apache.commons.csv;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.StringReader;
import org.apache.commons.csv.ExtendedBufferedReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ExtendedBufferedReader_ESTest extends ExtendedBufferedReader_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      StringReader stringReader0 = new StringReader("org.apache.commons.csv.ExtendedBufferedReader");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.getLineNumber();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StringReader stringReader0 = new StringReader("org.apache.commons.csv.ExtendedBufferedReader");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.readAgain();
      assertEquals((-2), int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      ExtendedBufferedReader extendedBufferedReader1 = new ExtendedBufferedReader(extendedBufferedReader0);
      int int0 = extendedBufferedReader1.read();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.read((char[]) null, 0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringReader stringReader0 = new StringReader("ZqGNS&4XuzH[6u6>f");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      ExtendedBufferedReader extendedBufferedReader1 = new ExtendedBufferedReader(extendedBufferedReader0);
      int int0 = extendedBufferedReader1.lookAhead();
      assertEquals(90, int0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      String string0 = extendedBufferedReader0.readLine();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      StringReader stringReader0 = new StringReader("/k8TiWn?Q>Zf~");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      String string0 = extendedBufferedReader0.readLine();
      assertEquals("/k8TiWn?Q>Zf~", string0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:47:45 GMT 2023
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
      StringReader stringReader0 = new StringReader("=Qoip&2=@l!");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.lookAhead();
      assertEquals(61, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      StringReader stringReader0 = new StringReader("Q");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.getLineNumber();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StringReader stringReader0 = new StringReader("Z");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.readAgain();
      assertEquals((-2), int0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      StringReader stringReader0 = new StringReader("=Qoip&2=@l!");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      int int0 = extendedBufferedReader0.read();
      assertEquals(61, int0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringReader stringReader0 = new StringReader("=Qoip&2=@l!");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      ExtendedBufferedReader extendedBufferedReader1 = new ExtendedBufferedReader(extendedBufferedReader0);
      String string0 = extendedBufferedReader1.readLine();
      assertEquals("=Qoip&2=@l!", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      StringReader stringReader0 = new StringReader("l");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      char[] charArray0 = new char[0];
      int int0 = extendedBufferedReader0.read(charArray0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      StringReader stringReader0 = new StringReader("");
      ExtendedBufferedReader extendedBufferedReader0 = new ExtendedBufferedReader(stringReader0);
      String string0 = extendedBufferedReader0.readLine();
      assertNull(string0);
  }
}
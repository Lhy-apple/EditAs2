/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 12:59:52 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.File;
import java.net.URL;
import java.util.Date;
import org.apache.commons.cli.TypeHandler;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeHandler_ESTest extends TypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Object object0 = TypeHandler.createValue((String) null, (Object) null);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class class0 = TypeHandler.createClass("");
      assertNull(class0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Number number0 = TypeHandler.createNumber("L`9Ae");
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      File[] fileArray0 = TypeHandler.createFiles("");
      assertNull(fileArray0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      TypeHandler typeHandler0 = new TypeHandler();
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      URL uRL0 = TypeHandler.createURL("Unable to find: ");
      assertNull(uRL0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      File file0 = TypeHandler.createFile("\"3");
      assertEquals(0L, file0.lastModified());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Object object0 = TypeHandler.createValue("...", class0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Class<String> class0 = String.class;
      Object object0 = TypeHandler.createValue("", class0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Date date0 = TypeHandler.createDate("KOY-B7IB");
      assertNull(date0);
  }
}

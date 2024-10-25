/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:48:03 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      Number number0 = TypeHandler.createNumber((String) null);
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
      URL uRL0 = TypeHandler.createURL(",JbYg@m>zyr^Cn");
      assertNull(uRL0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createFile((String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.File", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        TypeHandler.createValue((String) null, class0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Class<String> class0 = String.class;
      Object object0 = TypeHandler.createValue("?b#@k?W;YaZ", class0);
      assertEquals("?b#@k?W;YaZ", object0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Date date0 = TypeHandler.createDate("W&;4Dy$GZHUJ<KEu!h");
      assertNull(date0);
  }
}

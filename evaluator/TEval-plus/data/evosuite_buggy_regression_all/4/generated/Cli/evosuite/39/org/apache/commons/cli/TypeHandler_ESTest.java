/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:06:13 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import org.apache.commons.cli.TypeHandler;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeHandler_ESTest extends TypeHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createValue("", (Object) "");
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // java.lang.String cannot be cast to java.lang.Class
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<?> class0 = TypeHandler.createClass("org.apache.commons.cli.ParseException");
      assertEquals("class org.apache.commons.cli.ParseException", class0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createFiles("org.apache.commons.cli.TypeHandler");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Not yet implemented
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Object object0 = TypeHandler.createValue("org.apache.commons.cli.TypeHandler", class0);
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      try { 
        TypeHandler.createURL("Unable to parse the URL: ");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unable to parse the URL: Unable to parse the URL: 
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createDate("org.apache.commons.cli.TypeHandler");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Not yet implemented
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      File file0 = TypeHandler.createFile("");
      assertFalse(file0.isHidden());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<String> class0 = String.class;
      Object object0 = TypeHandler.createValue("", class0);
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Object object0 = TypeHandler.createValue(":%=9,SmWgVulu&0", class0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      try { 
        TypeHandler.createNumber("");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // For input string: \"\"
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      try { 
        TypeHandler.createNumber("org.apache.commons.cli.ParseException");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // For input string: \"org.apache.commons.cli.ParseException\"
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }
}
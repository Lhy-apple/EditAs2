/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:28:18 GMT 2023
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
        TypeHandler.createValue("1scluS", (Object) "1scluS");
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
      try { 
        TypeHandler.createClass("orgapache.cNmmons.cl.PatternO'tionBuilder");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unable to find the class: orgapache.cNmmons.cl.PatternO'tionBuilder
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.openFile((String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.mock.java.io.MockFileInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createFiles("");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Not yet implemented
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeHandler typeHandler0 = new TypeHandler();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      try { 
        TypeHandler.createURL("!$1-;f_Z");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unable to parse the URL: !$1-;f_Z
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        TypeHandler.createDate("");
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Not yet implemented
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      File file0 = TypeHandler.createFile("");
      assertEquals("", file0.getName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Object> class0 = Object.class;
      try { 
        TypeHandler.createValue("d!@7gTA4H|A},`1Z", class0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unable to find the class: d!@7gTA4H|A},`1Z
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<String> class0 = String.class;
      String string0 = TypeHandler.createValue("8", class0);
      assertEquals("8", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Integer integer0 = TypeHandler.createValue("", class0);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      try { 
        TypeHandler.createNumber("1 I@Z9gV{+ U");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // For input string: \"1 I@Z9gV{+ U\"
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      try { 
        TypeHandler.createNumber("<q}.3y?r6^SM7:ytw*=");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // For input string: \"<q}.3y?r6^SM7:ytw*=\"
         //
         verifyException("org.apache.commons.cli.TypeHandler", e);
      }
  }
}
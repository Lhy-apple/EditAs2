/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:02:32 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.lang.LocaleUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocaleUtils_ESTest extends LocaleUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocaleUtils.availableLocaleSet();
      Set set0 = LocaleUtils.availableLocaleSet();
      assertFalse(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LocaleUtils localeUtils0 = new LocaleUtils();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      List list0 = LocaleUtils.localeLookupList((Locale) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("vx_XU_#`i");
      assertNotNull(locale0);
      assertEquals("XU", locale0.getCountry());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale((String) null);
      assertNull(locale0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("ru");
      assertEquals("", locale0.getCountry());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("ve_FU");
      assertEquals("ve", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("16%L!");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 16%L!
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("~>#R>JmiH45@*8L5");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ~>#R>JmiH45@*8L5
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("x9");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: x9
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("y|cl8tx,`:.{V`|(");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: y|cl8tx,`:.{V`|(
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("fm~G9wUd{s_'");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: fm~G9wUd{s_'
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ve_<m,P");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ve_<m,P
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ve_u<m,P");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ve_u<m,P
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("vk_K3gum,P");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: vk_K3gum,P
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ve_Khgum,P");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ve_Khgum,P
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ve_FUKhum,");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ve_FUKhum,
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Locale locale0 = new Locale(".C", ".C", ".C");
      List list0 = LocaleUtils.localeLookupList(locale0, locale0);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      List list0 = LocaleUtils.localeLookupList(locale0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      Locale locale1 = Locale.CANADA;
      List list0 = LocaleUtils.localeLookupList(locale0, locale1);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      boolean boolean0 = LocaleUtils.isAvailableLocale((Locale) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      List list0 = LocaleUtils.languagesByCountry((String) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String string0 = "TH";
      LocaleUtils.languagesByCountry("TH");
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("TH");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: TH
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      List list0 = LocaleUtils.countriesByLanguage((String) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      List list0 = LocaleUtils.countriesByLanguage(".C");
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      List list0 = LocaleUtils.countriesByLanguage("th");
      assertEquals(1, list0.size());
  }
}